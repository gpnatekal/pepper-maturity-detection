import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { ToastrService } from 'ngx-toastr';

export const authGuard: CanActivateFn = () => {
  const router = inject(Router);
  const toastr = inject(ToastrService)
  
  const token = sessionStorage.getItem('authtoken');
  
  if (!token) {
    toastr.error('Please login again.' , 'Session expired!', {
      timeOut: 4000,
      positionClass: 'toast-bottom-center'
    })
    router.navigateByUrl('/login');
    return false;
  }

  return true;
};
